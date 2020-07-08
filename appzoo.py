from flask import Flask, request, render_template
from joblib import dump, load

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('homezoo.html')

	
@app.route('/prediction', methods=['POST','GET'])
def prediction():
	if request.method=='POST':
		loaded_model = load('Zoo.joblib')
		classes = ["Thú", "Chim", "Bò Sát","ĐV dưới nước","Lưỡng Cư","Côn Trùng","ĐV Không Xương Sống"]
		X_new = ([[request.form['long_hair'], 
			request.form['long_vu'],
			request.form['egg_'],
			request.form['milk_'],
			request.form['Airborne'],
			request.form['Aquatic'],
			request.form['an_thit'],
			request.form['co_rang'],
			request.form['xuong_song'],
			request.form['ho_hap'],
			request.form['noc_doc'],
			request.form['co_canh'],
			request.form['so_chan'],
			request.form['duoi'],
			request.form['song_bay_dan'],
			request.form['catsize'],
			]])
		y_new = loaded_model.predict(X_new)
		return render_template('resultzoo.html',results=classes[y_new[0]-1])
	
if __name__ == '__main__':
	app.run()
	