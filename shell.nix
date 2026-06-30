{
  pkgs ? import <nixpkgs> { },
}:
let
  python-derivation = pkgs.python314.withPackages (
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

      pybind11
      pybind11-stubgen
      setuptools
      pip
    ]
  );
in
pkgs.mkShell {
  packages = with pkgs; [
    python-derivation
    ffmpeg
    xxd
  ];

  shellHook = ''
    export CPATH="$CPATH:${python-derivation}/include/python3.14"
  '';
}
