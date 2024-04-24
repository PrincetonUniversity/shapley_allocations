import unittest

class TestExample(unittest.TestCase):
    #Note that every test starts with the prefix "test_".
    def setUp(self) -> None:
        """
        This function is called before every individual test.
        """
        self.a = 5
        self.b = 3
    
    def test_add(self):
        #Suppose my code generates a+b
        my_result = self.a + self.b #Imagine my_result is generated by my code
        expected_result = 8 #This is the result I expect to get for this test
        self.assertEqual(my_result, expected_result)
    
    def test_sub(self):
        #Suppose my code approximates a-b and I know the test should be positive for some reason
        my_result = self.a - self.b #Imagine my_result is generated by my code
        expected_bound = 0 #This is the lower bound I expect
        self.assertGreater(my_result, expected_bound)
    
    def test_this_fails(self):
        #Example to what it looks like when a test fails
        self.assertEqual(0, 1, "This test will always fail.")