import ezsmdeploy

ezonsm = ezsmdeploy.Deploy(model='model.pickle',
                           script='modelscript_ezsm.py',
                           requirements=['numpy', 'tensorflow', 'pickle'])


