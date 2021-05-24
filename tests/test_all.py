# -*- coding: utf-8 -*-
import glob
import os
import unittest

import pandas as pd
from fferm import fferm,clustering


class TestFferm(unittest.TestCase):
    """Advanced test cases."""

    def test_index_multi(self):
        l = [1,2,2]
        x = 2
        expected = [1,2]
        actual = fferm.index_multi(l,x)
        self.assertEqual(expected,actual)

    def test_overlap_index(self):
        l = [1,2,2]
        expected = [1,2]
        actual = fferm.index_multi(l)
        self.assertEqual(expected,actual)

    def test_fferm(self):
        movie_path = 'test.mp4'
        output_path = 'face_images'
        output, known_face_encodings = fferm.fferm(movie_path,output_path)
        files = glob.glob(os.path.join(output_path,'fr_ndex','*'))
        assert files > 0
        assert isinstance(output,pd.DataFrame)
        assert isinstance(known_face_encodings,list())


if __name__ == '__main__':
    unittest.main()
