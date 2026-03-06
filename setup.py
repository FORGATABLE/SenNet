from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="nnUNet",
        packages=["SenNet", "nnunetv2"],  # 显式列出所有包
    )
