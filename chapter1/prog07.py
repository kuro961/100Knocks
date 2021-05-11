def get_text(x: int, y: str, z: float):
    return f'{x}時の{y}は{z}'

def main():
    x = 12
    y = "気温"
    z = 22.4
    print(get_text(x, y, z))

if __name__ == '__main__':
    main()