## TensorFlow troubleshooting

If the app shows a TensorFlow import error (DLL load failed), try the following steps:

1. Make sure you're running Streamlit with the same Python interpreter where you installed the packages. Example:

```cmd
cd /d "D:\Stock Market Fundamental Analysis"
py -3.12 -m streamlit run app.py
```

2. Install the Microsoft Visual C++ Redistributable (x64) if not already installed:

- Download and run `vc_redist.x64.exe` from Microsoft:
	https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

3. If you expect to use GPU acceleration, install the correct NVIDIA drivers, CUDA toolkit, and cuDNN versions compatible with your TensorFlow release. Verify `nvidia-smi` is available on PATH.

4. Run the in-app diagnostic (in Technical Analysis) — it will show Python info, TF import traceback, MSVC DLL checks, and CUDA/NVIDIA info to help identify the root cause.

5. If problems persist, collect the full traceback shown by the diagnostic and search the TensorFlow issues page or open a new issue with the traceback:
	https://github.com/tensorflow/tensorflow/issues

