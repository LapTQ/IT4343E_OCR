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


mục tiêu đạt được là cty bảo hiểm sẽ nhập liệu lại hồ sơ khách hàng từ hồ sơ scan
từ các loại giấy tờ như: CMND, CCCD, Giay khai sinh...
đọc hóa đơn thuốc, bệnh án ..
nghĩa là giờ mình cần nhanh mới kịp
cái nhập liệu này nó chỉ làm 1 lần rồi thôi.

theo đối tác POC. họ cần 200 mẫu để tranning
vậy mình 2k mẫu chắc chất hơn nhỉ :D


1. Kiểm tra giải pháp OCR giấy tờ tùy thân hiện có của các bên
2. Nếu oke --> Cung cấp tập mẫu Hồ sơ yêu cầu bảo hiểm (HSYCBH) để training
3. Tiến hành POC cho phần OCR HSYCBH

đây là 1 y/c phía khách hàng
vậy là mình làm được cái 1
xong nó sẽ thuê mình làm cái 2, 3
cái số 2-3 đó là RPA của mình hướng tới




State-of-the-Art in Action: Unconstrained Text Detection

Character Region Awareness for Text Detection

A Survey of Deep Learning Approaches for OCR and Document Understanding

Sach a TA (hoi nghi nay chuyen sau ve text)

https://viblo.asia/p/deep-learning-key-information-extraction-from-document-using-graph-convolution-network-bai-toan-trich-rut-thong-tin-tu-hoa-don-voi-graph-convolution-network-djeZ1yPGZWz

https://ai.googleblog.com/2022/04/formnet-beyond-sequential-modeling-for.html



What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis



Detecting dense text in natural images

MMOCR: A Comprehensive Toolbox for Text Detection, Recognition and Understanding

Donut: Document Understanding Transformer without OCR


[https://viblo.asia/p/bai-toan-trich-xuat-thong-tin-tu-hoa-don-ORNZqd4nK0n](https://viblo.asia/p/bai-toan-trich-xuat-thong-tin-tu-hoa-don-ORNZqd4nK0n)

[https://viblo.asia/p/trich-xuat-thong-tin-bang-bieu-cuc-don-gian-voi-opencv-1Je5E7M1ZnL](https://viblo.asia/p/trich-xuat-thong-tin-bang-bieu-cuc-don-gian-voi-opencv-1Je5E7M1ZnL)

[https://viblo.asia/p/information-extraction-trong-ocr-la-gi-phuong-phap-nao-de-giai-quyet-bai-toan-yMnKMjzmZ7P](https://viblo.asia/p/information-extraction-trong-ocr-la-gi-phuong-phap-nao-de-giai-quyet-bai-toan-yMnKMjzmZ7P)

[https://stackoverflow.com/questions/67763853/text-recognition-and-restructuring-ocr-opencv](https://stackoverflow.com/questions/67763853/text-recognition-and-restructuring-ocr-opencv)

[https://www.youtube.com/watch?v=fswR5cbmq-c](https://www.youtube.com/watch?v=fswR5cbmq-c)



## References

https://keras.io/examples/vision/handwriting_recognition/
https://keras.io/examples/audio/ctc_asr/
https://github.com/TomHuynhSG/vietnamese-handwriting-recognition-ocr/blob/main/Vietnamese_Handwritten_Recognition_CRNN.ipynb
https://arthurflor23.medium.com/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16
https://github.com/huyhoang17/Vietnamese_Handwriting_Recognition
https://github.com/pbcquoc/vietnamese_ocr
https://github.com/pbcquoc/vietocr/blob/master/vietocr_gettingstart.ipynb
https://arxiv.org/pdf/1905.05381.pdf
