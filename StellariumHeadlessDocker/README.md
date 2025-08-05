Assuming you've gotten this folder with the Dockerfile, stellarium.ini, and entrypoint.sh,
you have all the files necessary to build the image and run the container (Stellarium headless,
makes the API available to the local / host machine, which is what we will use to get the training images
for the visual transformer star/planet/moon filter).

Docker must be installed on your machine of course.
Commands:

1.  docker build -t stellarium-headless .
2.  docker run -d -p 8090:8090 --name stellarium stellarium-headless

The API should now be available to the image gathering scripts, on local port 8090.
You can now do a basic curl command test to see that the API is indeed available:

3.  curl http://localhost:8090/api/main/status

In order to see more detailed information on container startup:

4.  docker logs stellarium
