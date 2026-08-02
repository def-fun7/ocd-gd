{
  description = "Fully automated native environment for AGAMA (Python 3.13 compatible | MacOS & Linux) along uv2nix";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # 1. Custom AGAMA build (Kept exactly as you had it)
        agama-python = pkgs.python3Packages.buildPythonPackage {
          pname = "agama";
          version = "latest";
          pyproject = true;

          src = pkgs.fetchFromGitHub {
            owner = "GalacticDynamics-Oxford";
            repo = "AGAMA";
            rev = "master";
            sha256 = "sha256-hj6kXimbPLjsJJGegenK7vENVYol5cx/Dm1vWA6fWn8=";
          };

          build-system = with pkgs.python3Packages; [
            setuptools
            wheel
          ];

          nativeBuildInputs = [
            pkgs.gsl
            pkgs.gmp
            pkgs.openblas
            pkgs.eigen
          ];

          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            scipy
            matplotlib
          ];

          doCheck = false;

          preBuild = ''
            echo 'import sys; sys.argv.append("--yes")' | cat - setup.py > setup.py.tmp && mv setup.py.tmp setup.py
          ''
          + pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
            export CFLAGS="-I${pkgs.gsl}/include -I${pkgs.openblas}/include -I${pkgs.eigen}/include/eigen3"
            export LDFLAGS="-L${pkgs.gsl}/lib -L${pkgs.openblas}/lib"
          '';
        };

        # 2. Parse uv.lock workspace via uv2nix
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # Construct package set from uv.lock
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Combine default build systems + python packages
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            python = pkgs.python313;
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                (final: prev: {
                  # Inject AGAMA into the uv2nix Python package set
                  agama = agama-python;
                })
              ]
            );

        # Create Python environment containing all workspace dependencies + dev extras
        pythonEnv = pythonSet.mkVirtualEnv "ocd-gd-env" {
          ocd-gd = [ "dev" ]; # Pulls core dependencies + project.optional-dependencies.dev
        };

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.gnumake
            pkgs.uv
          ];

          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            echo "========================================================="
            echo "🪐 AGAMA + uv2nix Automated Environment Loaded"
            echo "========================================================="
            echo "Platform detected: ${system}"
            echo "AGAMA is pre-compiled via Nix, while all dependencies are driven by uv.lock"

            ${pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
              export CFLAGS="-I${pkgs.gsl}/include -I${pkgs.openblas}/include"
              export LDFLAGS="-L${pkgs.gsl}/lib -L${pkgs.openblas}/lib"
            ''}
          '';
        };

        apps =
          let
            mkApp = scriptText: {
              type = "app";
              program = toString (
                pkgs.writeScript "app-script" ''
                  #!/usr/bin/env bash
                  export PATH="${pythonEnv}/bin:${pkgs.bash}/bin:$PATH"
                  ${scriptText}
                ''
              );
            };
          in
          {
            docs = mkApp "make -C docs html";
            makeReq = mkApp "uv export --format requirements-txt > requirements.txt && uv clean";
          };
      }
    );
}
