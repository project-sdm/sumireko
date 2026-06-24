{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    (python314.withPackages (
      ps: with ps; [
        numpy
        fastapi
        fastapi-cli
        opencv4
        matplotlib
      ]
    ))
  ];
}
