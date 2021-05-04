For CS 5984 Security Analytics project work

It is an experiment to check whether adversarial training is transferable in the text domain. So, 
the basic idea is if we train a model adversarially on adversarial samples from attack A, does it
make the model more robust against attack B?

To run this project, you need to do the following things:
1. Create a file **config.yml** in the project root directory and copy the **config.yml.demo** file contents to the new file
2. Change the file paths if need be    
3. Open the main.py (entry point) file and change the TODOs according to the following
   a. For training, set the **task** variable to "train"
       i. if you want to train your model normally, set **at** to False, specify the normal training parameters
   in Args constructor
       ii. if you want to train your model normally, set **at** to True, specify the adversarial training parameters
   in Args constructor
   
   b. For generating adversarial samples, set the **task** variable to "pre-generate", set **at** to False,
   specify the parameters for pre-generate in Args constructor, specify the model path that you want to attack
   
   c. For evaluating model's performance, set the **task** variable to "evaluate", set **at** to True or False
   depending on whether you want to evaluate a normal model or an adversarial model.
   

Reference: https://github.com/QData/TextAttack