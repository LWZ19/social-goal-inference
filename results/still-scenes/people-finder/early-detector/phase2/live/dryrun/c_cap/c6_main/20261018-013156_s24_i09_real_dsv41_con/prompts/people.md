## system
You help a mobile robot that works among people to see who is around it. From its camera image you list the people in view and say, for each of them, where they are and which way they face. Where each person stands on the floor is worked out from your points afterwards.
## task
The image that follows is the robot's current camera view. List every person visible in it, each one once, also those who are far away, partly hidden or seen from behind. List at most {max_people} people. If nobody is visible, give an empty list.

For each person, give a few words that pick them out, a point on their body, a point on what they face, and the way they face.

"appearance": a few words that pick this person out from the others in the image, such as what they wear or carry.

"facing": the way the person faces as seen in the image. Left and right are those of the image, not the person's own.
- towards_camera: facing the camera; you see their front.
- towards_right: facing partly the camera and partly the right edge of the image.
- right: facing the right edge of the image; you see their side.
- away_right: facing partly away from the camera and partly the right edge of the image.
- away_from_camera: facing away from the camera; you see their back.
- away_left: facing partly away from the camera and partly the left edge of the image.
- left: facing the left edge of the image; you see their side.
- towards_left: facing partly the camera and partly the left edge of the image.
- unclear: you cannot tell which way they face.
## reasoning
Before the list, fill in "analysis": how many people you see, where each of them is in the image, and what shows the way each of them faces.
## point_format
Set "body_point" to a point on the middle of the person's body: on the torso, well inside their visible outline, not on the head, not on the legs and not on the floor at their feet. Set "faced_point" to a point on what the person faces, another person or a thing, well inside its visible outline, if that is visible in the image; set it to null if it is not. {point_instruction}
## output
Answer with one JSON object that matches the schema, and nothing else.
