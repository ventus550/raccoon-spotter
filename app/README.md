# Raccoon Spotter

## Quick Start 
 
- Quick run with Docker:
```shell
$ docker run --rm -p 5000:5000 ghcr.io/wenjie-hoo/raccoon-spotter:latest
``` 
- Go to http://localhost:5000 and enjoy :tada: 
 
Screenshot: 
 
<p align="center">
  <img src="./docs/screenshot.avif" alt="raccoon">
</p> 

------------------ 
 
## Run with Docker 
 
#### Use prebuilt image 
 
```shell
$ docker run --rm -p 5000:5000 ghcr.io/wenjie-hoo/raccoon-spotter:latest
```
 
#### Build locally 
 
With **[Docker](https://www.docker.com)**, you can quickly build and run the entire application in minutes :whale: 

```shell 
$ cd raccoon-spotter/app/
 
$ docker build -t raccoon-spotter .
 
$ docker run -it --rm -p 5000:5000 raccoon-spotter
``` 
#### Load local model 
```shell 
docker run -p 5000:5000 \
           -e MODEL_PATH="/app/models/your_model_name.keras" \
           -v /path/to/your/local/model/directory:/app/models \
           -d raccoon-spotter
``` 
Open http://localhost:5000 and wait till the webpage is loaded. 

## Local Installation 

It's easy to install and run it on your machine:  
Place your model in the `models/` folder, named `trained_model.keras`.
 
```shell 
$ cd raccoon-spotter/app/
 
$ pip install -r requirements.txt 
 
$ python app.py 
``` 
 
#### Accessing the application

Open a web browser and navigate to http://localhost:5000/ to access the Raccoon Spotter application. Upload an image using the web interface and wait for the model to process and display the results.

## API Usage
#### <strong>Endpoint:</strong> `/predict`  
<strong>Method:</strong> POST  
<strong>Request Body:</strong> JSON containing a base64 encoded image string.

```json
{ "image": "base64_encoded_image_data" }
``` 

<strong>Response:</strong> JSON object with the processed image encoded in base64.

```json
{ "image_base64": "processed_image_data" }
``` 


