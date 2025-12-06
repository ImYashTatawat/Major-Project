First draft of python file for making base of our project of extracting intrested images base on our given prompts like person ,car or food anything which can be searched in the combination of "OR" and "AND" between keywords.
This model works on pretrained model from YOLOv5 of COCO dataset which comprises 80 different objects and 330K images with annotations and bounding boxes.
We would try to make this more precise for what we are giving it prompts which can be understood by the model to give us our intrest of resuts. we would try to add up other models also whic can be give us more precise results by filtering out the 
resulted images based on our prompt meanwhile we also try to work on its front-end.

we used pretrained model for this because this is the best what available for our project(may find more best and useful models as project develops, it is not feasible FOR NOW to train a different model. due to plenty of resoources and time.
we may look at it later. For now we try to compile best tools to make our project.





for our final BIG model 
I have uploaded the synonyms,newnlp_parsr,build_synonyms,backend2,loop files . synonyms file has all the list of objects 600+, newnlp_parser is responsible for matching query with dictionary and image findings. run all the files in single directory except build synonyms (run it only when using different model other than yolov8m_ovi7.pt =>open images pretrained 600+ categories medium model ) then run streamlit run loop.py in terminal and search, as you lowe the confidence score it scand each image withsmaller  detecing boxes resulting in discovering of more objects in image. if we increase threshold then the findings will be one object or none. try to download model first from site of ultralytics .
