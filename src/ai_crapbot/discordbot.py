import discord
import requests
import torch
import random
import os
import discordimgbot

client = discord.Client()

# Define the chatbot model
class Chatbot(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Chatbot, self).__init__()
        self.hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.output = torch.nn.Linear(input_size + hidden_size, output_size)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.hidden(combined))
        output = self.softmax(self.output(combined))
        return output, hidden

# Define the training function
def train(input_tensor, target_tensor, model, optimizer, criterion):
    hidden = torch.zeros(1, hidden_size)
    optimizer.zero_grad()
    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i].unsqueeze(0))

    loss.backward()
    optimizer.step()

    return loss.item() / input_tensor.size(0)

# Define the data preprocessing function
def preprocess_data(message, keywords):
    input_tensor = torch.zeros(len(message), len(keywords))

    for i, word in enumerate(message):
        for j, keyword in enumerate(keywords):
            if keyword in word.lower():
                input_tensor[i][j] = 1

    return input_tensor


# Define the response function
def generate_response(input_tensor, model, keywords, responses):
    hidden = torch.zeros(1, hidden_size)

    for i in range(input_tensor.size(0)):
        output, hidden = model(input_tensor[i], hidden)

    _, topi = output.topk(1)
    response_index = topi.item()

    return random.choice(responses[keywords[response_index]])

# Define the Discord event handlers
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    # Check if message contains an image attachment
    if message.attachments and message.attachments[0].url.endswith((".jpg", ".jpeg", ".png", ".gif")):
        # Get the URL of the image attachment
        image_url = message.attachments[0].url

        # Save the image to a local directory
        directory = "images"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, 'discordfile.jpg')
        with open(filename, "wb") as f:
            f.write(requests.get(image_url).content)

        # Generate a response based on the image
        message.channel.send(discordimgbot.returnimage('discordfile.jpg'))

        # Send the response and the saved image back to the Discord channel


    # Define the keywords and corresponding responses
    keywords = ["hello", "goodbye", "thanks"]
    responses = {
        "hello": ["Hi there!", "Hello!", "Greetings!"],
        "goodbye": ["Goodbye!", "Farewell!", "Bye!"],
        "thanks": ["You're welcome!", "No problem!", "Anytime!"]
    }

    # Preprocess the message
    input_tensor = preprocess_data(message.content.split(), keywords)

    # Generate a response
    if input_tensor.sum() > 0:
        response = generate_response(input_tensor, model, keywords, responses)
        await message.channel.send(response)

# Define the model parameters and training loop
input_size = len(keywords)
hidden_size = 128
output_size = len(keywords)
learning_rate = 0.01
num_epochs = 1000

model = Chatbot(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.NLLLoss()

# Generate training data
training_data = []
for i in range(1000):
    keyword = random.choice(keywords)
    message = keyword + " " + " ".join([random.choice(["some", "random", "words"]) for j in range(random.randint(1, 10))])

    # Generate input and target tensors
    input_tensor = preprocess_data(message.split(), keywords)
    target_tensor = torch.LongTensor([keywords.index(keyword)])

    training_data.append((input_tensor, target_tensor))

# Train the model
for epoch in range(1, num_epochs + 1):
    random.shuffle(training_data)
    total_loss = 0

    for input_tensor, target_tensor in training_data:
        loss = train(input_tensor, target_tensor, model, optimizer, criterion)
        total_loss += loss

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: {total_loss / len(training_data)}")
