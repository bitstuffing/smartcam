# SmartCAM

This is simple. A python3 program, based on configuration parameters.

This program analyzes a local or remote stream (f.i. based on IoT) and provide events (f.i. routines, automatic welcome to your house, alarms...).

This is a base to enrich your own assistant.


## Optional requirements

If you want, you're able to use with a local webcam. 

Launch the simple attachment:

```python
python server.py
```
You should change your configuration for production environment in config.cfg file.

You will be able to see the detection in console and in a window.

## install.sh

This script will automatically install dependencies and download trained models to get this program working fine without worry you. 

It's a requirement to run it before launch main proccess. If you want to use your own trained models, this step is not necessary.

```bash
chmod +x install.sh
./install.sh
```

## Run it

Simply run:

```python
python3 main.py
```

## License

Released under Creative Commons 4.0 license, developed by @bitstuffing with love.

This development is based on OpenCV and his license, and Haar or YOLOv3 models have his own licences.
