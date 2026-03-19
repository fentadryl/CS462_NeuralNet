# To compile this code in Google Cloudshell

# Clone Repository
git clone https://github.com/fentadryl/CS462_NeuralNet.git

# Move into the project folder
cd CS462_NeuralNet/myneuralnet

# This creates a 'bin' folder and compiles your code into it
mkdir -p bin
javac -d bin src/*.java

# Ececute Code
java -cp bin myneuralnet.neuralnet
