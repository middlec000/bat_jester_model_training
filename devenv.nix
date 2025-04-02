{ pkgs, ... }:
{
  name = "bat_jester_model_training";

  packages = [
    pkgs.git
    pkgs.zlib
    pkgs.ruff
  ];

  languages.python = {
    enable = true;
    version = "3.9";
    venv = {
      enable = true;
      requirements = ''
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
        tensorflow_hub
        pywavelets
      '';
    };
    uv.enable = true;
  };
}
