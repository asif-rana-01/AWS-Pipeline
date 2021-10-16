from pickle import loads

def load_model(modelpath):
    clf = loads(os.path.join(modelpath, 'model.pickle'))
    return clf

def predict(model, payload):
    try:
        # in remote / container based deployment, payload comes in as a stream of bytes
        out = [np.frombuffer(payload[0]['body']).reshape((1,64))]
    except Exception as e:
       out = [type(payload),str(e)] # useful for debugging!

    return out

# Test
if __name__ == '__main__':
    model = load_model('')
