import unittest

import bioregistry
import pandas as pd

from glycowork.glycan_data.loader import build_custom_df, df_glycan


class TestSemantics(unittest.TestCase):
    def setUp(self) -> None:
        self.df = build_custom_df(df_glycan, kind="df_tissue")

    def test_tissue_ids(self):
        """Test that entries in the tissue_id column are valid CURIEs."""
        for index, tissue_id in self.df["tissue_id"].iteritems():
            with self.subTest(index=index, tissue_id=tissue_id):
                self.assertTrue(pd.notnull(tissue_id))
                self.assertIn(
                    ":",
                    tissue_id,
                    msg=f"Tissue identifier not written as a CURIE: {tissue_id}",
                )
                prefix, identifier = tissue_id.split(":")
                self.assertIsNotNone(
                    bioregistry.normalize_prefix(prefix),
                    msg="prefix is not registered in the Bioregistry",
                )
                pp = bioregistry.get_preferred_prefix(prefix)
                self.assertEqual(
                    pp,
                    prefix,
                    msg=f"prefix is not standardized with the Bioregistry. Should be: {pp}",
                )
                pattern = bioregistry.get_pattern(prefix)
                self.assertIsNotNone(pattern, msg="")
                self.assertRegex(
                    identifier,
                    pattern,
                    msg=f"local unique identifier does not match standard pattern from the Bioregistry: {pattern}",
                )
