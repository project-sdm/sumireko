{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    ffmpeg
    (python314.withPackages (
      ps: with ps; [
        numpy
        fastapi
        fastapi-cli
        opencv4
        librosa
        python-multipart
        faiss
      ]
    ))
  ];

  env = {
    PYTHONPATH = ".";
  };
}
