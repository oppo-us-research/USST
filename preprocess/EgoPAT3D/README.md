
## Note
To preprocess the raw RGB-D recordings (`.mkv` format) from Azure Kinect camera, the python package `pyk4a` needs to be installed on Windows operation system where a monitor needs to be connected. 

### Install the pyk4a on Windows (monitor required)
 - Download and install the Azure Kinect SDK from [Azure Kinect SDK 1.4.1.exe](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md).
 - Setup two Environment Variables: 1) put the path `C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin` in the `PATH` variable. 2) create a new Variable `CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1`.
  - Install Anaconda installer (Windows version), and create a virtual python environment: 
    ```shell
      conda create -n usst python=3.7
      conda activate usst
    ```
  - [Optioinal] If you encountered problem related to the missing VC++ Build Tools when installing pyk4a by pip as belows, try install Visual Studio Build Tools from [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  - Install the python package `pyk4a` by the command:
    ```shell
      pip install pyk4a --no-use-pep517 --global-option=build_ext --global-option="-IC:\Program Files\Azure Kinect SDK v1.4.1\sdk\include" --global-option="-LC:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\lib"
    ```

### Install FFMpeg:
   - Download FFMpeg binary installer, and set the environment variable `PYTHONPATH=C:\Users\${your_name}\ffmpeg\bin\ffmpeg.exe`.
