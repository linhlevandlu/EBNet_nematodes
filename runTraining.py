
from cnnmodel3_2018 import train

#
DATA=['v11','v12','v13','v14','v15','v16','v17','v18']
#DATA=['v10']
FTRAIN_FIX = '/data3/linhlv/Tete_dec/i250x200/csv/train_'
FTEST_FIX = '/data3/linhlv/Tete_dec/i250x200/csv/test_'
SAVE_FIX = '/data3/linhlv/Tete_dec/Output/train_losses/loss_5000eps_rate82_0001_Early_'
epochs = 5000
for i in DATA:
	ftrain = FTRAIN_FIX + i+'.csv'
	ftest = FTEST_FIX + i+'.csv'
	savemodel = SAVE_FIX + i+'.pickle'
	saveloss = SAVE_FIX + i+'_loss.jpg'
	savetest = SAVE_FIX + i+'_test.jpg'
	print(ftrain)
	print(ftest)
	print(savemodel)
	print(saveloss)
	print(savetest)
	train(ftrain,ftest,epochs,savemodel,saveloss,savetest)

print("Finish!!")
