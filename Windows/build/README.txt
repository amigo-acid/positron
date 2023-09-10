Thank You for using version 1.0 of the windows version of positron.

Pre-Requisites: Python 3.0 must be installed.

Usage: 
Both Posture detection and Blink counter is supported. Upon running the executable 2 windows will open representing both the features. 
To start posture detection, maintain perfect posture then press the 'p' key. An audio alert plays when posture deteriorates. 
The blink counter automatically starts up and gives information on time since the application was opened, number of times the user has blinked and the blinks per minute so far. 
You can compare this blinks per minute with the average 10 - 20 to understand if you are straining your eyes and need a break.
In order to stop, press 'q' on both windows.

Note: 
Eye counter works much better for users without using glasses, with glasses please maintain proper lighting and posture for accurate results.
When starting posture detection, ensure you do not go into bad posture mode for a period of 3 seconds until the landmarks and connections are drawn in the window.
To adjust the sensitivity of posture, edit the value of the int in line 236 currently set to 10, increasing the value will decrease the range at which bad posture is detected.
To adjust the sensitivity of blink counter, edit the value of blink_threshold var in the python file line 67, increasing the value will decrease the sensitivity.