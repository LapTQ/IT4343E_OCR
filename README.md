=========================================
TÁCH VÀ ALIGN PHẦN GIẤY TRONG ẢNH SCAN





















tìm hiểu tại sao ở bước Permute lại làm mô hình không học được
untraced function
scan ảnh trước để xem kích thước ảnh thế nào rồi resize cố định về h= 118, tại sau cùng thì chỉ có 1 neuron
variable length, phần này chắc là height của ảnh là cố định 118 rồi, còn chiều dài thì tự tính được
tìm hiểu mạng CNN khác hiệu quả hơn
thay thế LSTM bằng attention xem sao?
tạo metric WER để early stopping tại điểm tốt nhất
preprocessing layers
augmentation






## References

https://keras.io/examples/vision/handwriting_recognition/
https://keras.io/examples/audio/ctc_asr/
https://github.com/TomHuynhSG/vietnamese-handwriting-recognition-ocr/blob/main/Vietnamese_Handwritten_Recognition_CRNN.ipynb
https://arthurflor23.medium.com/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16
https://github.com/huyhoang17/Vietnamese_Handwriting_Recognition
https://github.com/pbcquoc/vietnamese_ocr
https://github.com/pbcquoc/vietocr/blob/master/vietocr_gettingstart.ipynb
https://arxiv.org/pdf/1905.05381.pdf
