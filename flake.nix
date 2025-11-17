{
  description = "Bat Jester Model Training Environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          name = "bat_jester_model_training";
          buildInputs = [
            pkgs.git
            pkgs.zlib
            pkgs.ruff
            pkgs.uv
            pkgs.cacert
            # C++ standard library (required by numpy)
            pkgs.stdenv.cc.cc.lib
            # OpenCV headless dependencies (no Qt/GUI)
            pkgs.libGL
            pkgs.glib
            pkgs.libglvnd
          ];
          shellHook = ''
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            
            # Set LD_LIBRARY_PATH for numpy and OpenCV
            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
                pkgs.zlib
                pkgs.stdenv.cc.cc.lib
                pkgs.libGL
                pkgs.glib
                pkgs.libglvnd
              ]
            }:$LD_LIBRARY_PATH
          '';
        };
      }
    );
}
