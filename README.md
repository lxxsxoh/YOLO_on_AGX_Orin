YOLO_on_AGX_Orin
Start without Docker
====================
Install Ultralytics Package
---------------------------
1. Update pakages list, install pip and upgrade to latest
<pre>
<code>
   sudo apt update
   sudo apt install python3-pip -y
   pip install -U pip
</code>
</pre>
3. Install ultralystics pip package with optional dependencies
<pre>
<code>
  pip install ultralystics
</code>
</pre>
- error
  <pre>
  <code>
     ERROR: scipy 1.10.1 has requirement numpy<1.27.0, >=1.19.5, but you'll have numpy 1.17.4 which is incompatible
     ERROR: ...
  </code>
  </pre>
3.1 Reinstall numpy
<pre>
<code>
   pip install numpy==1.24.4
</code>
</pre>
3.2 Reinstall ultralystics pip package with optional dependencies
<pre>
<code>
  pip install ultralystics
</code>
</pre>
4. Reboot the device
<pre>
<code>
  sudo reboot
</code>
</pre>
Install PyTorch and Torchvision
-------------------------------
Python 3.8.10

PyTorch = 2.1.0a0+41361538.nv23.06

Torchvision = 0.16.2+c6f3977
