#!/usr/bin/env python3
"""Serve the two plate_sliding Host9 policies (experiment 20261027_v1) with serve_host9_real."""
import logging

import serve_host9_real as base

base.VERSION = '20261027_v1'
base.PORTS = {8004: ('plate_sliding', 'raw6'), 8005: ('plate_sliding', 'encoder')}
base.PROMPTS = {**base.PROMPTS, 'plate_sliding':
                'Slide the pregrasped pink plate through the white rigid guide and lift it clear; keep holding it'}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    base.main()
