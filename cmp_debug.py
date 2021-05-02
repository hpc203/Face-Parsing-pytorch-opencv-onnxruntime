import numpy as np

torch_input = np.load('torch_input.npy')
torch_out = np.load('torch_out.npy')

cv_input = np.load('cv_input.npy')
cv_out = np.load('cv_out.npy')

onnxrun_input = np.load('onnxrun_input.npy')
onnxrun_out = np.load('onnxrun_out.npy')

if np.array_equal(torch_input, cv_input):
    print('pytorch和opencv的输入是相同的')
else:
    print('pytorch和opencv的输入的平均差是', np.mean(torch_input - cv_input))

if np.array_equal(torch_input, onnxrun_input):
    print('pytorch和onnxruntime的输入是相同的')
else:
    print('pytorch和onnxruntime的输入的平均差是', np.mean(torch_input - onnxrun_input))

if np.array_equal(cv_input, onnxrun_input):
    print('opencv和onnxruntime的输入是相同的')
else:
    print('opencv和onnxruntime的输入的平均差是', np.mean(cv_input - onnxrun_input))

if np.array_equal(torch_out, cv_out):
    print('pytorch和opencv的输出是相同的')
else:
    print('pytorch和opencv的输出的平均差是', np.mean(torch_out - cv_out))

if np.array_equal(torch_out, onnxrun_out):
    print('pytorch和onnxruntime的输出是相同的')
else:
    print('pytorch和onnxruntime的输出的平均差是', np.mean(torch_out - onnxrun_out))

if np.array_equal(cv_out, onnxrun_out):
    print('opencv和onnxruntime的输出是相同的')
else:
    print('opencv和onnxruntime的输出的平均差是', np.mean(cv_out - onnxrun_out))