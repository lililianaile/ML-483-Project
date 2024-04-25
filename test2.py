import torchvision.transforms as transforms
from torchvision.datasets import EMNIST # import EMNIST dataset
from torchvision.transforms import ToTensor, Normalize # import ToTensor for Preprocessing the dataset
import matplotlib.pyplot as plt # for visualisation
import numpy as np # used for arrays
import gym # import gym which is used for creating the environment

train_dataset = EMNIST(root='data',split='letters', train = True, download = True, transform = transforms.Compose([
   transforms.ToTensor(),  # Convert image to tensor
   transforms.Normalize((0.5,), (0.5,))  # Normalize image data
]))# download training data in data-folder

test_dataset = EMNIST(root='data', split='letters', train = False, download = True, transform = transforms.Compose([
   transforms.ToTensor(),  # Convert image to tensor
   transforms.Normalize((0.5,), (0.5,))  # Normalize image first close to 0 and between [-1,1]
])) # download test data in data-folder

# Example of accessing a sample from the dataset
print("Enter sample index:")
sample_index = int(input()) # Take input from user to show visuals of a sample index
image, label = train_dataset[sample_index]  # normally labeled datasets are used for supervised learning, but here we're using EMNIST to simplyfiy 
print("The label of the Sample is:", label) # prints lable of the sample index (Label means the letter of the alphabeth e.g. Label = 23 is equal to Letter = W)
letters_array = np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])
print("The lable:",label, " is equivalent to letter:", letters_array[label-1])
print("Sample data:", image)

def visualEMNIST(imageAsArray):
   imageAsArray = imageAsArray.reshape(28,28) # 28 x 28 pixels
   plt.imshow(imageAsArray, cmap = 'gray') 
   plt.show()

visualEMNIST(image)

# set up own environment
gym.register( 
    id='HandwrittenLetterRecognition-v0', 
    entry_point='test2:HandwrittenLetterRecognitionEnv' # change path to correct file name in the end!
)



class HandwrittenLetterRecognitionEnv(gym.Env):
   def __init__(self, train_dataset):
      self.train_dataset = train_dataset
      self.num_states = 28  # Assuming images are 28x28 pixels = 784
      self.num_actions = 26  # Assuming 26 letters in the alphabet -> find other actions maybe look thru array values or sth
      self.sequence_length = 28 # Number of pixels the agent reviews at the same time
      self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.sequence_length, self.num_states), dtype=np.uint8) 
      self.action_space = gym.spaces.Discrete(self.num_actions) # those define format of valid actions and observations
      self.current_state = None
      self.current_index = 0
      # batch or sequence to look at all 784

   def step(self, action):
      # Execute action on the current letter
      letter_image = self.train_dataset[self.current_index]
      reward = self.calculate_reward(action, letter_image)
      next_state = letter_image
      # Move to the next letter in the dataset
      self.current_index = (self.current_index + 1) % len(self.train_dataset)
      # Determine if the episode is done
      done = self.current_index == 0 # this means we iterated though all letters

      return next_state, reward, done
   
   #visualize
   def render(self, mode='human'):
      current_image = self.train_dataset[self.current_index].reshape(28,28)
      plt.imshow(current_image, cmap = 'gray')
      plt.show()

   #reset environment to initial state
   def reset(self):
      self.current_index = 0
      return self.train_dataset[self.current_index]
   

   def calculate_reward(self, action, letter_image):
      predicted_letter = chr(action + 65) # convert action index to ascii charcter (A to Z)
      true_letter = chr(np.argmax(letter_image)+65) # convert true label index to ascii charcter (A to Z)

      if predicted_letter == true_letter:
         reward = 1 # positive reward for correct prediciton
      else:
         reward = -1 # negativ ereward for wrong prediciton

      return reward

class QLearningAgent:
   def __init__(self, num_states, num_actions, learning_rate = 0.1, discount_factor = 0.9, epsilon = 0.1 ):
      self.num_states = num_states
      self.num_actions = num_actions
      self.learning_rate = learning_rate
      self.discount_factor = discount_factor
      self.epsilon = epsilon
      self.q_table = np.zeros ((num_states, num_actions))

   def choose_action(self,state):
      if np.random.uniform(0,1) <self.epsilon:
         #Explore: choose a random action
         return np.argmax(self.q_table[state])
      else:
         #Exploit: choose the action with the highest Q-value
         return np.argmax(self.q_table[state])
      
   def update_qtable(self, state, action, reward, next_state):
      # Q Learning update rule
      current_q_value = self.q_table[state, action]
      max_next_q_value = np.max(self.q_table[next_state])
      new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
      self.q_table[state, action] = new_q_value



#initialize environment
env = HandwrittenLetterRecognitionEnv(train_dataset)
agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)



#Training Loop
num_episodes = 5
for episode in range(num_episodes):
   state = env.reset()
   done = False
   total_reward = 0

   while not done:
      # Choose action
      action = agent.choose_action(int(state))
      # take action
      next_state, reward, done, _ = env.step(action)
      # update Q-table
      agent.update_qtable(state, action, reward, next_state)

      #env.render()
      # update state
      state = next_state
      # Accumulate total reward
      total_reward += reward

# print total reward for the episode

print("Episode{}: Total reward={}".format(episode+1, total_reward))



# implement: use test data to check accuracy and visualize it"""

 
