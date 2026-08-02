{
  description = "Fully automated native environment for AGAMA + ocd-gd (Python 3.13, uv2nix-driven)";

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

        # Parse pyproject.toml + uv.lock
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Patch AGAMA with required native C libraries and build tools
        agama-python = final: prev: {
          agama = prev.agama.overrideAttrs (old: {
            # C dependencies for linking
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.gsl
              pkgs.gmp
              pkgs.openblas
              pkgs.eigen
            ];

            # Python build system dependencies
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              final.setuptools
              final.wheel
              final.numpy
              final.scipy
              final.matplotlib
            ];

            # Pre-append '--yes' flag to non-interactive AGAMA setup script
            preBuild = (old.preBuild or "") + ''
              echo 'import sys; sys.argv.append("--yes")' | cat - setup.py > setup.py.tmp && mv setup.py.tmp setup.py
            '';
          });
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            python = pkgs.python313;
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                agama-python
              ]
            );

        pythonEnv = pythonSet.mkVirtualEnv "ocd-gd-env" workspace.deps.all;
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
            echo "🪐 AGAMA + uv2nix Environment Loaded"
            echo "========================================================="
            echo "Platform detected: ${system}"
            echo "pyproject.toml + uv.lock are the source of truth."
          '';
        };

        apps =
          let
            mkApp = scriptText: {
              type = "app";
              program = toString (
                pkgs.writeShellScript "app-script" ''
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
