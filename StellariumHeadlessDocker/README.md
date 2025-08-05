Assuming you've gotten this folder with the Dockerfile, stellarium.ini, and entrypoint.sh,
you have all the files necessary to build the image and run the container (Stellarium headless,
makes the API available to the local / host machine, which is what we will use to get the training images
for the visual transformer star/planet/moon filter).

1.  Docker must be installed on your machine of course.
Commands:
a.  docker build -t stellarium-headless .
b.  docker run -d -p 8090:8090 --name stellarium stellarium-headless

The API should now be available to the image gathering scripts, on local port 8090.
You can now do a basic curl command test to see that the API is indeed available:
c.  curl http://localhost:8090/api/main/status

In order to see more detailed information on container startup:
d.  docker logs stellarium
