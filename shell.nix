{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    (python314.withPackages (
      ps: with ps; [
        faiss
        fastapi
        fastapi-cli
        librosa
        nltk
        numpy
        opencv4
        psycopg
        psycopg-pool
        python-multipart
      ]
    ))
    ffmpeg
    xxd
  ];
}
