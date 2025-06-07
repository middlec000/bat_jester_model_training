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
          ];

          # Set SSL certificate path
          shellHook = ''
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          '';
        };
      }
    );
}
