# pylint: disable=invalid-name,too-many-locals
import hashlib


class HashedDict(dict):
    """Similar to dict, but can fetch keys using partial sha1 of the key


    To get a value, one can specify a key, or its sha1 hash string. Partial strings are also supported, provided
    that they are unique enough. If a partial string results in ambiguity, a KeyError is raised.
    Getting values using hash strings has the complexity of O(n), where n is the number of items in the dict

    """

    def real_key(self, key):
        """ Return the real key that corresponds to `key`

        If `key` is the real key, return itself. If `key` is a sha1 has substring, return the key that corresponds
        to it. If no such a key present, or if the substring is not unique enough, raise KeyError
        """
        if key in self.keys():
            return key
        else:
            key_length = len(key)
            keys = [k for k in self.keys() if hashlib.sha1(k.encode('utf-8')).hexdigest()[0:key_length] == key]
            if len(keys) == 1:
                return keys[0]
            elif len(keys) > 1:
                msg = "Non unique key '%s'" % key
                raise KeyError(msg)
            else:
                raise KeyError(key)

    def __getitem__(self, key):
        return dict.__getitem__(self, self.real_key(key))
