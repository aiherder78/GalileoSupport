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
Note:  You can see the question/answer sessions with Perplexity in the Perplexity.ai folder.  For the initial conversation, I just copy/pasted into a text file and no longer have the session link (I don't yet have a Perplexity account, maybe that'll change soon).  For the second one, I printed the html page to a PDF file from within the browser.  It has one of the input fields sitting over some of the text (for later files, I used developer tools in the browser to remove that field before printing).  I may just upload the most relevant two session files to this repo.
TODO:  upload Perplexity question/answer files.

None of this was immediate, it was a long series of questions submitted to Perplexity (almost sounds like an ad, but I'm not paid and I have definitely seen a massive improvement in directed search by having an LLM on top of a search engine like that).  I asked a ton of questions around astronomy concerning what maps or software I could use, and once I'd settled on Stellarium how to show only what you'd likely see with just your eyes or a normal camera (didn't even know about how star magnitude worked), and then a ton of questions on how to compare known star positions to images.  There are probably further modifications I'll have to make since if you want a CNN to be realtime, you're likely going to use YOLO CNNs.

The Galileo folks think this is infeasible, so if it works I'm just going to casually introduce it in a meeting and hopefully after they find out there are tons of people submitting data sightings using it.  Expensive hardware with lots of processing in a pipeline means you likely have few monitored points, while there's a massive number of UFO/UAP enthusiasts worldwide and I bet a ton of them have computers.  If you have a systematic and collaborative network using a high level of authoritative knowledge and SLAM / thermal in an autonomous fashion, probably the only thing else you'd need would be some blockchain.  Most earlier government attempts at explanations like swamp gas or everything being "Venus" wouldn't get close to being believable as the software would have already filtered known objects and likely (depending on distance) have quite a bit more data on position, velocity, shape, and heat emission levels of any given object.  If we combine that with realtime ADS-B (Automatic Dependent Surveillance-Broadcast) from the net, then we'd also filter out known aircraft (and potentially drone as well) transponders, at least those going by the rules.

For this project, I am using Ubuntu 22.04 and Python version > 3.10.

------------------------
TODO:  Include instructions for setting up a Python virtualenv
TODO:  Include a requirements file for pip install (this will get bigger as I go)
TODO:  Keep updating the requirements file as I have further required Python modules with more project updates

Instructions / Steps:

Installing Stellarium, starting it and accessing the http server API (to test with the browser):
(No, I never remember everything and in this case have never used Stellarium before - I have just read a ton about it and here's my question to Perplexity for these instructions:
"While I'm waiting for the camera to arrive (and getting ready to order the other one), I need to setup the web server to get http API functionality with Stellarium.  Do you know the steps for that on Ubuntu?" - in this case I'm moving toward duplicating the steps to get training images from Stellarium for training a CNN...and yeah, I have already ordered the P2 Pro / included Kevin's referral ID on the order link).

From the Ubuntu terminal (ctrl-alt-t):  (added by me since I like to get extremely detailed in steps)

sudo apt-get install stellarium-qt6

Launch Stellarium and navigate to Configuration > Tools > Plugins.
Configure the plugin settings:
  Set the desired port number (default is 8090)
  Choose whether to allow connections from other machines
Restart Stellarium for the changes to take effect.
Access the HTTP API by opening a web browser and navigating to:
  http://localhost:8090/api/main/status
Replace "localhost" with your machine's IP address if accessing from another device.
You can now use HTTP GET requests to control Stellarium remotely. For example:
  http://localhost:8090/api/main/focus?target=Mars

Remember to check Stellarium's documentation for a full list of available API commands and their usage.
(Thanks Perplexity, but I'm asking you first anyway...I'll check the documentation if something still doesn't work or if I need to double check if something might be available that you didn't mention)

TODO:  Include any needed Stellarium config files / location in this project's repo after I add them

---------------
Setting up a basic Python script to access data:
(TODO:  Include this first Python script file name / location in this project repo after I add it)

