{ ... }:
let
  nixpkgsTarball = builtins.fetchTarball
    { url = https://github.com/NixOS/nixpkgs/archive/refs/tags/21.11.zip;
      sha256 = "162dywda2dvfj1248afxc45kcrg83appjd0nmdb541hl7rnncf02";
    };
  nixpkgs = import nixpkgsTarball { };
  pythonPackages = nixpkgs.python39Packages;

in
nixpkgs.mkShell rec {
  buildInputs = [
    pythonPackages.python
    pythonPackages.black
    nixpkgs.nodePackages.pyright
    nixpkgs.nodejs
  ];
}
