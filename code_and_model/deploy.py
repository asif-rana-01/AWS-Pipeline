from sagemaker.tensorflow import TensorFlowModel
import argparse
import pandas as pd
from util import preprocess_x, reshape

parser = argparse.ArgumentParser(
    description='Deploy a TensorFlow model to a SageMaker endpoint'
)
parser.add_argument('--model_data', help='Path to model in an S3 bucket',
                    required=True)
parser.add_argument('--role', help='IAM role that has SageMaker privileges',
                    required=True)
parser.add_argument('--instance_type', help='Instance size',
            TensorFlowModel        default='ml.c5.xlarge')

args = parser.parse_args()

model = (# entry_point='inference.py',
                        model_data=args.model_data,
                        role=args.role)
predictor = model.deploy(initial_instance_count=1,
                         instance_type=args.instance_type)

# Test data assumed to be a list of lists of raw SiPM values
test_data = {
    'instances': [
        [15.0, 18.0, 19.0, 16.0, 11.0, 26.0, 15.0, 30.0, 83.0, 138.0, 142.0, 122.0, 117.0, 112.0, 93.0, 108.0, 96.0, 91.0, 66.0, 113.0, 110.0, 102.0, 87.0, 86.0],
        [85.0, 87.0, 93.0, 96.0, 124.0, 128.0, 113.0, 100.0, 53.0, 66.0, 42.0, 246.0, 99.0, 82.0, 82.0, 65.0, 73.0, 86.0, 54.0, 80.0, 146.0, 88.0, 103.0, 82.0], 
        [63.0, 72.0, 176.0, 112.0, 96.0, 91.0, 75.0, 84.0, 45.0, 68.0, 165.0, 79.0, 98.0, 95.0, 78.0, 77.0, 66.0, 105.0, 239.0, 109.0, 142.0, 115.0, 90.0, 109.0]
    ]
}

# Ignore the following three preprocessing instructions if
# inference.py is specified in entry_point

# Create a DataFrame of the SiPM values
df = pd.DataFrame(test_data['instances'],
                  columns=[f'SiPM{i}'
                           for i in range(1,
                           len(test_data['instances'][0]) + 1)])

# Standard scale the input
X, _ = preprocess_x(df)

# Reshape the input based on the channel's spatial configuration
X = reshape(X, { 'num_channels': 8 }) # num_channels could be 8, 12,
                                       # or 24 for det 2
# Get predictions
predictions = predictor.predict(X)
