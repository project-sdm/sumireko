{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    (python314.withPackages (
      ps: with ps; [
        opencv-contrib-python
        fastapi
        fastapi-cli
      ]
    ))
  ];
}
