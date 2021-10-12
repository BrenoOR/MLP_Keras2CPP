#include <QCoreApplication>
#include "src/keras_model.h"

#include <iostream>

using namespace std;
using namespace keras;

// Step 1
// Dump keras model and input sample into text files
// python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet
// Step 2
// Use text files in c++ example. To compile:
// g++ keras_model.cc example_main.cc
// To execute:
// a.out

int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\n"
           << "Keras model will be used in C++ for prediction only." << endl;

  DataChunk *sample = new DataChunk2D();
  sample->read_from_file("../MLP/sample.dat");
  std::cout << "Depth: " << sample->get_3d().size() << std::endl;
  std::cout << "Rows: " << sample->get_3d().at(0).size() << std::endl;
  std::cout << "Columns: " << sample->get_3d().at(0).at(0).size() << std::endl;
  KerasModel m("../MLP/mlp.nnet", true);
  std::cout << "Input cols: " << m.get_input_cols() << std::endl;
  std::cout << "Input rows: " << m.get_input_rows() << std::endl;
  std::cout << "Output len: " << m.get_output_length() << std::endl;
  DataChunk *flat = new DataChunkFlat();
  vector< vector<float> > output;
  for (int i = 0; i < sample->getRows(); i++) {
      vector<float> row = sample->get_3d().at(0).at(i);
      flat->set_data(row);
      vector<float> rowVals = m.compute_output(flat);
      output.insert(output.end(), rowVals);
  }
  delete sample;

  int right = 0;
  int wrong = 0;

  DataChunk *sample_resps = new DataChunk2D();
  sample_resps->read_from_file("../MLP/sample_resps.dat");
  vector< vector<float> > resps = sample_resps->get_3d().at(0);

  int guess;
  bool guessRight;
  for (int i = 0; i < output.size(); i++) {
      for (int j = 0; j < output.at(i).size(); j++) {
          if (output.at(i).at(j) > 0) {
              guess = 1;
          } else {
              guess = 0;
          }

          if (resps.at(i).at(j) == guess) {
              guessRight = true;
          } else {
              guessRight = false;
          }
      }
      if (guessRight) {
          right++;
      } else {
          wrong++;
      }
  }

  std::cout << "\nGuessing:" << std::endl;

  std::cout << " Guess Right: " << right << " | Total: " << (right+wrong) << std::endl;
  std::cout << " Porcentagem: " << (float)(100*right)/(right+wrong) << "%\n";
  std::cout << " Guess Wrong: " << wrong << " | Total: " << (right+wrong) << std::endl;
  std::cout << " Porcentagem: " << (float)(100*wrong)/(right+wrong) << "%\n";

  return 0;
}
