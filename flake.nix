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

        # Parse pyproject.toml + uv.lock — this is now the single source of truth
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        agamaFixupOverlay = final: prev: {
          agama = prev.agama.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [ ])
              ++ [
                pkgs.gsl
                pkgs.gmp
                pkgs.openblas
                pkgs.eigen
              ]
              ++ [
                final.setuptools
                final.wheel
                final.numpy
                final.scipy
                final.matplotlib
              ];

            preBuild =
              (old.preBuild or "")
              + ''
                # Decline the optional CVXOPT and UNSIO libraries outright — they require
                # network access to download (blocked by the Nix sandbox) and ocd-gd
                # doesn't need Schwarzschild modelling or N-body snapshot I/O.
                # This must happen BEFORE the --yes hack below, since --yes would
                # otherwise auto-approve these downloads too.
                sed -i "/if ask('CVXOPT library/,/Y\/N\] '):/c\\
                if False:" setup.py
                sed -i "/if ask('UNSIO library/,/Y\/N\] '):/c\\
                if False:" setup.py

                echo 'import sys; sys.argv.append("--yes")' | cat - setup.py > setup.py.tmp && mv setup.py.tmp setup.py
              ''
              + pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
                export CFLAGS="-I${pkgs.gsl}/include -I${pkgs.openblas}/include -I${pkgs.eigen}/include/eigen3 ''${CFLAGS:-}"
                export LDFLAGS="-L${pkgs.gsl}/lib -L${pkgs.openblas}/lib ''${LDFLAGS:-}"
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
                agamaFixupOverlay
              ]
            );

        # "all" pulls in the dev optional-dependency group (pytest, sphinx, etc.)
        # as well as ocd-gd itself and its runtime deps.
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
