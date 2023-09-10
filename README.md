<p align="center">
  <a href="https://pose-detection-git-master-subhrajyotighosh972-gmailcom.vercel.app/"><img src="https://github.com/amigo-acid/positron/assets/62471257/5eeac271-be09-403f-be16-a98fe1c06dd1" alt="Logo" height=170></a>
</p>
<h1 align="center">positron</h1>



## What is positron?

positron is the working buddy for everyone spending hours upon hours hunched in front of computers, getting work done. It is an ergonomic healthcare app that employs computer vision to assess and enhance posture and blink patterns for those with sedentary lifestyles

There are currently 2 versions:
1. Web version:- Built using HTML, CSS and JS
2. Windows version:- Desktop application built using Python and OpenCV


Deployed at:
https://pose-detection-git-master-subhrajyotighosh972-gmailcom.vercel.app/

## Installation

Git clone https://github.com/amigo-acid/positron.git
Move into Windows/build and execute positron.exe

## Usage

### Web version
In the current web version, posture detection is supported. To use it, simply click 'Record Angle' on the web app, assume your desired posture you'd like to maintain, and then click 'Stop Recording.' You can continue using your computer as usual. If your posture deteriorates for more than 5 seconds, an audio alert will remind you to correct it

### Windows version
In the current Windows version, posture detection and blink counter is supported. Upon running the executable 2 windows will open representing both the features. To start posture detection, maintain perfect posture then press the 'p' key. Similar to the web version, audio alert plays when posture deteriorates. The blink counter automatically starts up and gives information on time since the application was opened, number of times the user has blinked and the blinks per minute so far. You can compare this blinks per minute with the average 10 - 20 to understand if you are straining your eyes and need a break. To stop, press 'q' on both windows.
