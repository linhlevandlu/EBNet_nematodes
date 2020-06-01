
from fineTune_freeze import fine_tune
'''
Fine-tuning the model 

'''
#def fine_tune(fmodel,ftrain,epochs,ftest,savemodel,saveloss,savetest):

# pre-train model
FMODEL = '/data3/linhlv/Tete_dec/model/pretrained_model/fine_tuning_on_facial_model_freeze_ts82_v10.pickle'
DATA=['v10','v11','v12','v13','v14','v15','v16','v17','v18']
#DATA=['v10']
FTRAIN_FIX = '/data3/linhlv/Tete_dec/i250x200/csv/train_'
FTEST_FIX = '/data3/linhlv/Tete_dec/i250x200/csv/test_'
SAVE_FIX = '/data3/linhlv/Tete_dec/Output/ft_train_losses/cnnmodel_8_outputs_fine_tune_10000epochs_lr01_00001_D_'
epochs = 10000
for i in DATA:
	ftrain = FTRAIN_FIX + i+'.csv'
	ftest = FTEST_FIX + i+'.csv'
	savemodel = SAVE_FIX + i+'.pickle'
	saveloss = SAVE_FIX + i+'_loss.png'
	savetest = SAVE_FIX + i+'_test.png'
	print(ftrain)
	print(ftest)
	print(savemodel)
	print(saveloss)
	print(savetest)
	fine_tune(FMODEL,ftrain,epochs,ftest,savemodel,saveloss,savetest)

print("Finish!!")
