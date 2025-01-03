# GalileoSupport
Searching the skies for anomalies - filtering out stars with the help of Stellarium and/or CNNs trained by its data (and using open source SLAM on thermal data to get relative position of heat-generating nearby flying objects).

Essentially, I'm thinking of using a CNN trained on Stellarium data (I'm not sure yet what the exact shape of the neural net will be since I need it to do CNNy things while also remembering time dependent and positional data - I'll likely try one with an added fully connected layer or two first) for star filtering and SLAM with two cheap consumer-level thermal cameras (the TOPDON TC001 and the P2 Pro).  Then once / if I get these both working to at least a minimal amount, I'll jam them together in one process with a CNN-SLAM approach and see how many people want to run it in order to have a distributed detection network (anybody).

For this project, I am using Perplexity.ai intensively in order to get over obstacles and answer questions.  I am a complete newbie to astronomy, but in trying to find a way to help the project, I think it would be a good idea to have an authoritative basis to filter out stars, planets, and potentially known meteorites.  While doing research with Perplexity, I mentioned that I'd seen papers (sometimes I trawl through arxiv.org) where satellites used convolutional neural nets (CNNs) to help determine their position in orbit and that I'd like to do the same thing in order to filter out stars in the night sky (in order to lower uncertainty that something is or is not an unknown craft).

Along the way, I also came up against my own budgetary constraints and lack of knowledge in various types of sensors, lack of anything more than a basic webcam (on my laptop - I like to use my computers and not have them sitting watching the sky all the time).  Once again, I asked Perplexity for help (in addition to a counter intel guy I found while watching documentaries on Leon Theremin and "The Thing" (spy device used by Russia to eavesdrop on a US ambassador in Moscow)).  He graciously listened to my idea and gave me feedback on sensors (thermal cameras) I might use.  Given that gmail and Youtube likely share data, I immediately next saw a review on Youtube for the P2 Pro and was pretty impressed.  With the encouragement from Kevin's email reply, I grabbed the P2 Pro and the TOPCON TC001 mentioned above.  With rush shipping, taxes, more cables (for the second thermal camera I'll probably need another $30 of cables), adapters, and a USB repeater (USB 2.0 length for a passive cable is 15 ft, but I'd like to leave the camera outside for long periods, so hopefully will send power), the price came to $716.53 total.  This all matters because the more I spend, the less people will be able to buy in.  Hopefully the price will come down further later (by tech advance and package deals becoming more frequent).

Just to hammer home how little I knew or had access to, by pestering the researcher who interviewed me to allow me onto the Galileo Project as a volunteer, I had gotten their YOLO-CNN model.  I didn't have any thermal cameras of my own, so I was left searching Youtube for FLIR night sky videos and downloaded two that were taken with (at least 2X magnification) hunting scopes.  Running the model against that yielded abysmal classification results, it had no idea what the stars were.  I think since the CNN was trained on their specific device, it needs those exact settings for good accuracy.  At least I got the inferrence working though - I hadn't worked directly with a YOLO model before and had to refit the weights after paying close attention to what I'd been told about their YOLO version, etc.

Early on in my research with Perplexity, it mentioned a project by a researcher who was attempting to use a CNN on night sky / visible stars and time data to get position on the Earth's surface at sea.  

He was able to get accuracy to within 6.5km (trained on a small slice of Earth's surface / night sky time co-relation for that, so it might not have been that accurate elsewhere).  This was my inspiration for the star filter portion of this project.  I figured that if the CNN could do that, then it could of course remember where the stars should be at any given time and be used as a simple filter.  Of course Stellarium also has planets and meteorites, but for the meteorites I'll likely have to run later checks using the same feedback mechanism used to train the CNN.  Essentially, Stellarium has an http server that provides an API interface you can use with a python module to direct Stellarium what to look at and what the camera parameters should be.  By generating images using that into a folder, you can then train the CNN (naming the images with the time data included - you give the time data to the neural net with the image in training).
This is the github repo for that researcher's project:
https://github.com/gregtozzi/deep_learning_celnav

Note:  You can see my question/answer sessions with Perplexity in the Perplexity.ai folder (these aren't all of them, only the ones I remembered to save, though likely there's enough there to give you a good summary of the progress toward the structure of the project).  For the initial conversation, I just copy/pasted into a text file and no longer have the session link (I don't yet have a Perplexity account, maybe that'll change soon).  For the second one, I printed the html page to a PDF file from within the browser.  It has one of the input fields sitting over some of the text (for later files, I used developer tools in the browser to remove that field before printing).

None of this was immediate, it was a long series of questions submitted to Perplexity (almost sounds like an ad, but I'm not paid and I have definitely seen a massive improvement in directed search by having an LLM on top of a search engine like that).  I asked a ton of questions around astronomy concerning what maps or software I could use.  Once I'd settled on Stellarium I asked how to show only what you'd likely see with just your eyes or a normal camera (didn't even know about how star magnitude worked).  Then I asked a ton of questions on how to compare known star positions to images.  There are probably further modifications I'll have to make since if you want a CNN to be realtime, you're likely going to use YOLO CNNs.

Expensive hardware with lots of processing in a pipeline means you likely have few monitored points, while there's a massive number of UFO/UAP enthusiasts worldwide and I bet a ton of them have computers.  If you have a systematic and collaborative network using a high level of authoritative knowledge and SLAM / thermal in an autonomous fashion, probably the only thing else you'd need would be some blockchain to reduce malicious actors' ability to corrupt data/change past feeds.  Most earlier attempts at explanations like swamp gas or everything being "Venus" wouldn't get close to being believable as the software would have already filtered known objects and likely (depending on distance) have quite a bit more data on position, velocity, shape, and heat emission levels of any given object.  If we combine that with realtime ADS-B (Automatic Dependent Surveillance-Broadcast) from the net, then we'd also filter out known aircraft (and potentially drone as well) transponders, at least those going by the rules.

TODO:  Advertise for interested UFO/UAP enthusiasts who have computers and are willing to buy some thermal cams.
Note:  There is something deeply philosophical and spiritual about being the center of your personal inner world while being not even a speck in a vast and awesome universe.  The potential passion getting hands-on with a feedback loop of this nature could inflame very likely would carry one through the hard times of life.  Even if we ALL never find anything in this search focus, for one to suddenly become aware of what little we know about our tiny corner of the universe and all the exploration options could be the spark needed.  This is a gift.  Barring direct instruction from Avi Loeb on any given thing subject to my contract as a volunteer, my work is public domain.

For this project, I am using Ubuntu 22.04 and Python version > 3.10.

------------------------

TODO:  Include instructions for setting up a Python virtualenv

TODO:  Include a requirements file for pip install (this will get bigger as I go)

TODO:  Keep updating the requirements file as I have further required Python modules with more project updates

Instructions / Steps:

Installing Stellarium, starting it and accessing the http server API (to test with the browser):
(No, I never remember everything and in this case have never used Stellarium before - I have just read a ton about it and here's my question to Perplexity for these instructions:
"While I'm waiting for the camera to arrive (and getting ready to order the other one), I need to setup the web server to get http API functionality with Stellarium.  Do you know the steps for that on Ubuntu?" - in this case I'm moving toward duplicating the steps to get training images from Stellarium for training a CNN...and yeah, I have already ordered the P2 Pro / included Kevin's referral ID on the order link).

From the Ubuntu terminal (ctrl-alt-t):  (added by me since I like to get extremely detailed in steps - from this point, all commands are assumed to be simply typed in the terminal, then press enter)

Add the official Stellarium repository and update the apt catalogue to reflect its contents (two terminal commands):
~~~
sudo add-apt-repository ppa:stellarium/stellarium-releases
sudo apt update
~~~
Install Stellarium and make sure it's installed by issuing a command to see its version (two terminal commands):
~~~
sudo apt install stellarium
stellarium --version
~~~
Command to run stellarium:
~~~
stellarium
~~~
On my computer with the version it installed from the official repo (version 24.3), it detected my location to my local town and also updated the time in the program / view to my computer's / local time.  The view essentially matches what I see outside except for the fact that it's a slightly different location than the default location Stellarium uses for my town.  It looks like the "view" of the ground around you, building, and trees, etc, is the same regardless of where you tell it to set the ground observer's location.  Using the Python script below and gathered lat/long for New Delhi, I successfully changed the location for my Stellarium client from the terminal, but other than it being daytime over there, nothing else changed as far as ground objects.

To install the RemoteControl plugin to Stellarium:
From within Stellarium, move the mouse cursor to the bottom left of the screen.  Two transparent bars will come up, one across the bottom, and another that goes a little way up on the left.  Move the mouse cursor up the side bar that goes up on the left side of the screen.  Click on the "Configuration Menu (F2)".  This is somewhat not-descriptive because what you'll get by clicking that is NOT what you get by pressing F2.  Pressing F2 gives you a much less functional version that has nothing you can directly change.
From within the menu window that pops up when you click on the sidebar config icon with the mouse, go to the Plugins tab.  Find "Remote Control" in the list and click on the "load at startup" checkbox.  Then click on the "configure" button to the right of that checkbox.  Check the "server enabled" and the "enable automatically at startup" checkbox.  This will turn the server on and always do it every time you start Stellarium.  The server turned on for me as soon as I clicked the "enable server" checkbox because the below url wasn't responding, but when I clicked it, the browser reloaded to Stellarium's http server webpage.
http://localhost:8090

You can also see a more simple / json list of Stellarium http server API status information at:
http://localhost:8090/api/main/status
  
Replace "localhost" with your machine's IP address if accessing from another device.
You can now use HTTP GET requests to control Stellarium remotely. For example:
  http://localhost:8090/api/main/focus?target=Mars

------------------
Your first simple view retrieval Python script:
Note: Stellarium must be running (in this case on your localhost, if you put it on a server on the net, change localhost to the server's IP, etc).
Stellarium API docs:  https://stellarium.org/doc/24.0/remoteControlApi.html

About the coordinates used by Stellarium's API (local coordinates):  
https://www.youtube.com/watch?v=sT8JIn7Q_Fo
(this section of the Perplexity session q/a files will be maddening - in my defense, I often stay up all night doing research and the spot where I didn't even notice it was already using the http API and kept asking question it had already answered, it was probably about 5am and I hadn't slept)

Note:  Stellarium uses right ascension and Declination (from an origin at the "vernal equinox") elsewhere according to Perplexity, and the three local coordinates are also not altitude and azimuth angles - that video above may help explain the relationship - to convert altitude and azimuth angles to those three numbers:

P (the line length to the arbitrary point mentioned in the video) is set to 1 (a unit sphere - the spot  on the unit sphere is wherever the light from any individual star goes through the imaginary unit sphere to reach your eye / the observer's camera).

Using the terms from that youtube video link (theta and phi, which are the same as azimuth and elevation respectively):

Theta angle (azimuth) is the angle from 0 degrees (north).

Phi angle (elevation) is the angle from 0 degrees (straight up, or in the video, the Z axis direction).

Here's how you get the x, y, z parameters for the local coordinates used by the Stellarium http API.  If you don't set it, in Stellarium your location will be whatever the Stellarium default is of course, and I saw another Youtube video stating the time for the observer would be your computer's current time.

~~~
#Note:  0 altitude points at the horizon, 90 points straight up
#Note2: 0 azimuth points north, 90 east, 180 south, 270 west, 360 back north
x=cos(altitude)⋅cos(azimuth)
y=cos(altitude)⋅sin(azimuth)
z=sin(altitude)
~~~

Without further ado, here's a basic Python script:
~~~
import requests
url = "http://localhost:8090/api/main/view"
params = {"altAz": "[0.0001,0,1]"}
response = requests.post(url, data=params)
print("View set:", response.status_code)
~~~

If you have Stellarium's view client running in another window, your Stellarium observer camera should have jerked to the new given vector direction.
Note:  That researcher with the CNN training Stellarium feedback loop had installed Stellarium "headless" and controlled it all with scripts.  I'm going to initially try to do this interactively / not headless.

Here's a short Python script and curl command to use the Stellarium http API to set the observer location:

Curl command:
~~~
curl -d "latitude=34.0522" -d "longitude=-118.2437" -d "altitude=100" http://localhost:8090/api/location/setlocationfields
~~~
Python program:
~~~
import requests
# Define the base URL for the Stellarium API
url_base = "http://localhost:8090/api/location/setlocationfields"
# Function to set the location in Stellarium
def set_location(latitude, longitude, altitude):
    # Prepare the data to be sent in the POST request
    data = {
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude
    }
    # Send the POST request to set the location
    response = requests.post(url_base, data=data)
    # Check if the request was successful
    if response.status_code == 200:
        print("Location set successfully!")
    else:
        print(f"Failed to set location: {response.status_code}, {response.text}")
# Example usage
if __name__ == "__main__":
    # Set your desired latitude, longitude, and altitude
    latitude = 34.0522  # Los Angeles latitude
    longitude = -118.2437  # Los Angeles longitude
    altitude = 100  # Altitude in meters
    # Call the function to set the location
    set_location(latitude, longitude, altitude)
~~~

TODO:  Include any needed Stellarium config files / location in this project's repo after I add them

---------------

Here's a Python script to set the Stellarium time.  During training image generation, we're going to be using this like a machine gun while also changing the camera view vector around between time changes.

~~~
import requests

# Stellarium HTTP API configuration
STELLARIUM_HOST = "http://localhost:8090"
SET_TIME_ENDPOINT = "/api/main/time"

def set_stellarium_time(year, month, day, hour, minute, second):
    """
    Set the time in Stellarium using the HTTP API.

    Args:
        year (int): Year to set.
        month (int): Month to set.
        day (int): Day to set.
        hour (int): Hour to set (24-hour format).
        minute (int): Minute to set.
        second (int): Second to set.
    """
    url = f"{STELLARIUM_HOST}{SET_TIME_ENDPOINT}"
    
    # Payload for setting the time
    payload = {
        "time": {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second
        }
    }
~~~
Thermal camera source code examples:
https://github.com/leswright1977/PyThermalCamera
https://www.youtube.com/watch?v=PiVwZoQ8_jQ

My initial code for reading the thermal cameras (both of them, takes a single integer argument - for instance, to read the cam at /dev/video0 just do "python3 readcam.py 0" - if you have multiple thermal cameras connected, each will likely have two videoX numbers so that the first will have /dev/video0 and /dev/video1 and the second will have /dev/video2 and /dev/video3 - therefore to read from both cameras use two terminals and issue the command on each of them with one having argument 0 and the other argument 2) is in "readcam.py".
TODO:  Change the coloring to match the heat levels using full RGB with redder being hotter.  Up the apparent resolution for better visibility like he did.

Now, here's the plan, and it's VERY simple (somewhat, the most complex thing is going to be mucking about with the AI/neural net hyperparameters):
1.  We're going to first set the Stellarium view parameters to look as close to the arbitrary camera we end up using first.
2.  Then, using the Stellarium API, we're going to generate tons of images from Stellarium for a long period of time from our planned observation position.  We're going to do what we can to grab all the star, planet, etc names.  According to this reddit discussion, one of the comments stated that YOLOv8 supports multiple labels per box, so as long as my automatic labeller code (cross fingers, I'm going to write it) successfully captures point sources of light (probably something to do with adaptive thresholding as a feature detector) and successfully uses the camera vector to that pixel and queries Stellarium to return the potential object name, I should have a good/authoritative object type and definite object name.  This also creates the potential for always using Stellarium as a backup to the trained CNN (as long as we know the camera central vector, ground position on Earth, and current time - this creates a minor success condition for worst case project failure mode).
3.  We're also going to have a script to alternate between the realtime camera and Stellarium images overnight and maybe do some kind of GAN training.
I'm personally going to use this as an opportunity to learn about GANs more now that I have a much better piece of hardware (an NVidia RTX 3080) than I did the last time (when I was using a much older laptop CPU  that tested my patience and I ultimately stopped that project due to most of the time being spent waiting).  My aim here will be to learn different ways the faker portion (I don't remember offhand right now what it's called) might readily replace data augmentation and how to balance learning rates between the classifier and faker/generator.  I'll be training them both on both real world images and Stellarium images, and of course the classifier will try to tell me if it's real world while the generator will try to fake the classifier.  If I can get the generator to apply weather patterns as data augmentation, I could then use that to potentially make my ultimate CNN star filter / identifier more fault tolerant (a fault being clouds, fog, or other occlusions of expected stars).
4.  My main challenge as a programmer is going to be to develop methods to apply what I've learned from my last personal project, the YOLO Pygame Labeller (you can see that in another repo on this github ID too) to automatically label boxes in generated images, both real world camera observation and Stellarium screenshots, with classifiers by object type (and maybe also the name of the object).  I'm going to absolutely spray these boxes around every single image and I'm probably going to have at least hundreds of thousands of YOLO standard compliant label files total, one for each image generated.  I'm probably going to have to learn about total available filenames per disk partition index type along the way and potentially reformat my disk to handle more, we'll see.
5.  One of my personal desired requirements will be to automatically gather the camera's star magnitude limitations and then somehow use that information to make the CNN automatically more accurate.
6.  Finally, we'll be running CNN and/or YOLO training on the many generated training images and comparing accuracy results between different methods.

The final product I want out of this will be a set of steps almost guaranteed to work as long as you have some minimum resolution of camera.  I'll distribute my best model, but ultimately it'll be nice to have a set of steps for everyone to follow that gets their own trained model regardless of camera type so that even if you have no knowledge of CNNs, you can just follow the steps and be set in some known period of time.

Oh yeah - regardless of the results of the above, I'm also going to stick on an open source SLAM with the two thermal cameras and some ADSB querying / trig to try to determine if some light probably corelates to ADSB transponders.  If I get a good CNN/YOLO model, I'll stick them together in CNN-SLAM (and hopefully better, YOLO-SLAM).

TODO:  As I start adding files, rename this one to background.txt and replace with a simpler readme that has just the basic install / config / run steps with a short blurb on purpose at the top.  Make a separate license file to specify that this is public domain.

--- Update ---

Shot some outside video with the P2 Pro.  As far as the raw data I'm getting, it's two images with essentially the same data but different coloring.  The top is grayscale and the bottom is green.  Since I'm using cv2 capture directly from my computer, I don't have access to the app they provide.  I'm going to need to do my own color conversion.  Same for the TOPDON TC001.
No problem though, I still have the raw temperature data / raw data provided again in grayscale.
So these are going to be relatively close range.  The focal length is such that they're best for up close, but I'm going to try to capture nearby flying craft (or birds, etc) anyway to see what I can do in spite of that.
I went to Walmart and bought a few tiny mason jars and I'm going to cut a hole in the lids for the cords and then stick them outside nearby pointing at an angle upward.  I'll have to remember to bring them in until I figure out how to get around the potential damage from the sun to the sensor during the day (manual on one of the devices says not to point it directly at the sun as it could damage the sensor).
I read somewhere that the P2 Pro has a visual sensor in addition, but I'm not seeing a difference in the type of data between the two images provided by both of the thermal cameras (other than the P2 Pro has a slightly larger max temp of 1112F versus the TOPDON TC001's 1022F max).

Therefore, among other things (two more USB C to A charging 10 ft charging and extension cables - 2x $8 for a total of $16, and a 4 port USB hub - roughly $30), I bought an outdoor wifi visual camera from Walmart as well (it's starting to look like I wasted my money completely on this - there's all kinds of unnecessary and obstructive things going on with that camera such as mandatory streaming to/from the cloud, using corporate certificates to prevent consumer programming, etc - I may take it back even though it's opened - it was an Arlo).  We'll see if I can access it directly, otherwise I'll take it back (using an app isn't an option for me as I need to integrate all the data I grab).  I'm probably at about $100 in cables and well over $800 now (I didn't need to rush shipping to get the earlier cables before Christmas, probably should take my time quite a bit more to keep costs down).

Other - I lost quite a bit of my Perplexity threads while rebooting many times (Brave browser isn't asking to restore tabs, probably need to adjust settings) during the troubleshooting of the thermal cameras.  The readcam.py is the ultimate result of all that.  

TODO:  It needs further testing with a virtualenv to make sure all installs are included in the instructions.  For instance, I had serious problems accessing the P2 Pro at first due to ffmpeg not being installed.  I guess it doesn't get installed when you install opencv.  There was also a UDEV rule I added for the TOPDON TC001 + added my user to the video group.

Done:  Upgrade to Perplexity Pro (this is my favorite LLM so far and since it has an extensive ability to automatically search the net, it fulfills quite a few of my ideal system requirements for now).

--- Update ----

I am currently Q/A'ing Perplexity on background info I need to gather arbitrary camera type parameters and set the Ocular plugin with them (Ocular is a plugin that simulates camera parameters when generating the view), then also adjust sky brightness.  I'm going to have a script to run overnight that takes note of maximum magnitude stars the camera can see.  I'm also going to have an included auto sky brightness adjusting function that I'll use in other code as well.

Ultimately, once the settings are gathered, that should be enough for a good feedback loop in getting tons of images over a long period of time for a given observation position, then training a neural net with it.  Then we'll have a CNN, YOLO, or maybe realtime visual transformer network that can automatically recognize stars and planets from that position.

My hope is that I can get it near enough that I can directly compare the two images throughout the night (camera for one, Stellarium for the other as I'll have the times sync'd).  I'll do a pixel by pixel comparison with auto overall brightness adjustment sent to Stellarium.  I'll keep track of stars the camera can't see and their magnitudes, as well as the dimmest star the camera did see.  For this script, light points the camera sees but Stellarium does not have will be ignored.

Brightness adjusting Python function (this changes the limiting star magnitude - higher magnitude stars are dimmer...so far at my knowledge level, I think that the naked eye can roughly see to about 6.5 magntitude):
Note:  Perplexity just generated this function, I still need to test it.  So far I've tested the vector change script and the observer location change one.
~~~
import requests

# Stellarium server details
stellarium_url = "http://localhost:8090"

# Set the new sky brightness value (adjust as needed)
new_brightness = 6.5  # Range is typically 1 (brightest) to 8 (darkest)

# API endpoint for setting properties
endpoint = f"{stellarium_url}/api/main/property"

# Set the light pollution limiting magnitude
payload = {
    "id": "LightPollutionLimitingMagnitude",
    "value": new_brightness
}

response = requests.post(endpoint, json=payload)

if response.status_code == 200:
    print(f"Sky brightness adjusted to {new_brightness}")
else:
    print("Failed to adjust sky brightness")
~~~

Getting parameters from an arbitrary camera using a wall, a tape measure, and something to mark with (I told Perplexity to use duct tape):

To gather the necessary parameters of an arbitrary camera for Stellarium using basic tools like a tape measure, duct tape, and a wall, you can follow these steps:
Measuring the Field of View
Set up your camera on a tripod facing a flat wall.
Use duct tape to mark a straight horizontal line on the wall.
Take a photo of the wall, ensuring the taped line is visible from edge to edge in the frame.
Measure the actual length of the taped line on the wall.
Measure the distance from the camera to the wall.

Calculating Parameters:

Calculate the horizontal field of view angle using trigonometry:

Field of view angle = 2 * arctan (half of duct tape line length / distance to wall)

Focal length ≈ sensor width / (2×tan(field of view angle / 2))

Note:  Field of view is commonly abbreviated as FOV.
 
Note: You'll need to estimate the sensor width based on the camera type (e.g., 36mm for full-frame, 22.3mm for APS-C).
Count the pixels in the horizontal and vertical directions of the image to get the resolution.

----
Update:
I haven't been using my thermal cameras overnight yet.  It's been bothering me that I might fall asleep and leave them to be exposed and damaged by the sun.  So, I've been looking for a solution that will automatically turn the camera lense away from the sun and into a shaded area before daylight, all for cheap.
I've made some progress:
I found a pan-tilt two servo set with some supports.  I also have been looking into options for cheaply controlling the servos while having it on the network.  Along the way, I've come across some curious issues, such as the controller boards mostly being for smaller servos and requiring less power as well as not coming with the required power supplies.  After at least two hours searching for just the power supply, the closest wall adapter I could find that wasn't too expensive and wasn't the type that you just modified the voltage by hand (if you bumped one of those you'd ruin the driver board) had a max current of 3 amps...I'm just about to give up and go with a buck converter (not efficient).  So now I'm just finishing up on my understanding on how I'll connect the driver board with an Arduino or RPi.  The code for controlling the servos is pretty simple, will be using output from the Arduino or RPI GPIO headers (if I get an RPi Zero, I'll get one of the solderless headers to go with it) for the data line and will be sending the ground of the power supply through the ground on the Arduino or RPi to provide a common reference point...etc.  Overall, I'll hopefully keep this entire hardware update under $150...hopefully.
OK, I'm punting.  Instead of designing a overly complex solution that would consist of an adapter and a buck step-down converter, I've emailed the servo board manufacturer for power supply suggestions since they don't have one packaged with the product.  I'll wait.
