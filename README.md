The goal of the project was to see if a straight forward CNN would be able to perform Crowd Counting, without the use of a Density Map.
Multiple CNN's are made. 
A secondary goal was to implement techniques to improve the prediction of negative samples (i.e. that do not include persons yet might 
seem to, for a computer). An attempt was made to implement ROS and focal loss. However, the images had too many dimension for ROS to handle.
The focal loss was successfully implemented on dummy data yet could not be converted to work with TensorFlow.
In the future the focal loss could be fixed, as it does have potential.
