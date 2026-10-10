.PHONY: build test app clean

build:
	cmake -S level0_2d -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$$(python3 -m pybind11 --cmakedir)
	cmake --build build -j
test:
	pytest -q
app:
	streamlit run app/app.py
clean:
	rm -rf build
