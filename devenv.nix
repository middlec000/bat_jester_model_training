{
  pkgs,
  ...
}:
{
  packages = [
    pkgs.git
    pkgs.zlib
    pkgs.ruff
  ];
  languages.python = {
    enable = true;
    version = "3.9";
    venv.enable = true;
    venv.requirements = ''
      ruff
      numpy
      librosa
      pandas
      matplotlib
      ipykernel
      jupyterlab
      pip
      notebook
      skl2onnx
      onnxruntime
      tensorflow
      tensoflow-hub
    '';
    uv.enable = true;
  };
  enterShell = ''
    echo "Entering shell"
    code .
  '';
}
