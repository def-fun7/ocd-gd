{
  description = "Fully automated native environment for AGAMA (Python 3.13 compatible | MacOS & Linux)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Custom Nix package definition for AGAMA
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
        ocd-gd = pkgs.python3Packages.buildPythonPackage {
          pname = "ocd-gd";
          version = "0.1.0";

          pyproject = true;
          src = ./.;
          nativeBuildInputs = with pkgs.python3Packages; [
            uv-build
          ];
          build-system = with pkgs.python3Packages; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = [
            agama-python
          ]
          ++ (with pkgs.python3Packages; [
            numpy
            matplotlib
            plotly
            numba
          ]);

          doCheck = false;
        };
        pythonEnv = pkgs.python3.withPackages (ps: [
          agama-python
          ocd-gd
          ps.numpy
          ps.matplotlib
          ps.plotly
          ps.numba
          ps.astropy
          ps.rich

          ps.sphinx
          ps.sphinx-rtd-theme
          ps.myst-parser

          ps.pytest
          ps.pytest-cov
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.gnumake
            pkgs.uv
          ];

          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            echo "========================================================="
            echo "🪐 AGAMA Automated Environment Loaded"
            echo "========================================================="
            echo "Platform detected: ${system}"
            echo "AGAMA is pre-compiled and available globally in this shell."
            echo "Try launching a python terminal via python command and running: import agama"

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
