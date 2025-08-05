This repo is meant for supporting a public, open source SkyWatch system.

What this means:
1.  One aspect of the project is development of a star/planet filter (unassisted eyes/cameras - magnitude 6).
2.  A future aspect will be integrating those feeds (and thermal potentially) with SLAM.
3.  To include code for picking up ADSB with SDR.
4.  Potentially - passive radar with SDR.
5.  Code to interoperate with other stations to put together a networked view.

An example of how this system might operate:
A user would have two SDRs connected to directional antennas, one pointed at the surveilled area and the other at the FM or TV trasnmission tower (opportunistic transmitter used for passive radar - the user isn't transmitting, just capturing echoes by comparing this reference to the surveilled signal - AKA a bi-static passive radar).
Also running on the processing computer the SDRs are connected to would be an ADSB process.  It would take any captured ADSB packets and compare them to the echo locations from the passive radar.
Once these objects get within visual range, if at night, cameras (visual, IR, thermal, etc) would be used with the star/planet filter visual transformer.  This neural network can detect the observer's location to within 5-6.5km and tell you if an observed object is a star, planet, or moon (based on its training, dependent on the datetime - trained up to magnitude 6 objects).  This should filter out any claims that sightings are "Venus", etc.  Anything coming nearby and getting enough exposure to the camera would also start to be subject to SLAM (simultaneous localization and mapping) processing.
