class Quaternion:
    def __init__(
            self,
            w,
            i,
            j,
            k
        ):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def __add__(
            self,
            q
        ):
        return Quaternion(
            self.w + q.w,
            self.i + q.i,
            self.j + q.j,
            self.k + q.k
        )
    
    def __sub__(
            self,
            q
        ):
        return Quaternion(
            self.w - q.w,
            self.i - q.i,
            self.j - q.j,
            self.k - q.k
        )
    
    def __mul__(
            self,
            c
        ):
        return Quaternion(
            self.w * c,
            self.i * c,
            self.j * c,
            self.k * c
        )
    
    def __truediv__(
            self,
            c
        ):
        return Quaternion(
            self.w / c,
            self.i / c,
            self.j / c,
            self.k / c
        )
    
    def __rmul__(
            self,
            c
        ):
        return Quaternion(
            self.w * c,
            self.i * c,
            self.j * c,
            self.k * c
        )
    
    def __matmul__(
            self,
            q
        ):
        a0, a1, a2, a3 = self.w, self.i, self.j, self.k
        b0, b1, b2, b3 = q.w, q.i, q.j, q.k
        return Quaternion(
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0
        )
    
    def __iter__(self):
        return iter((self.w, self.i, self.j, self.k))
    
    def __abs__(self):
        return np.sqrt(self.w ** 2 + self.i ** 2 + self.j ** 2 + self.k ** 2)
    
    def __str__(self): 
        return f'[{self.w},{self.i},{self.j},{self.k}]' 

    def squared_norm(self):
        return self.w ** 2 + self.i ** 2 + self.j ** 2 + self.k ** 2

    def inv(self):
        return Quaternion(
            self.w,
            -self.i,
            -self.j,
            -self.k
        )