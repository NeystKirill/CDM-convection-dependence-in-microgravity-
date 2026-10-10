.PHONY: build build-l0 build-l1 test app clean

build: build-l0 build-l1

build-l0:
	cmake -S level0_2d -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$$(python3 -m pybind11 --cmakedir)
	cmake --build build -j
build-l1:
	cmake -S emulator -B build_l1 -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$$(python3 -m pybind11 --cmakedir)
	cmake --build build_l1 -j
test:
	pytest -q
app:
	streamlit run app/app.py
clean:
	rm -rf build build_l1
