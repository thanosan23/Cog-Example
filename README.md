# Cog Test
This an example using Replicate's cog api.

Cog allows you to be able to create a machine learning model, containerize the machine learning model using docker, and deploy the machine learning model to an endpoint quickly.

## How to use

First, create the model:
```
python3 main.py
```

Create the docker image using cog:
```
cog build -t linear 
```

Run the docker image:
```
docker run -p 3000:5000 --platform=linux/amd64 linear
```

The model will be deployed on `http://localhost:3000/predictions`. You can do a post request to this endpoint.

For example:
```
curl http://localhost:3000/predictions -X POST -H 'Content-Type: application/json' --data '{"input": {"x": "11.0"}}'
```

You can also do:
```
cog predict -i x=11
```
