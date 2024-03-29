/* -*- c++ -*- */
/*
 * Copyright 2011,2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

namespace gr {
namespace qtgui {

enum trigger_mode {
    TRIG_MODE_FREE,
    TRIG_MODE_AUTO,
    TRIG_MODE_NORM,
    TRIG_MODE_TAG,
};

enum trigger_slope {
    TRIG_SLOPE_POS,
    TRIG_SLOPE_NEG,
};

} /* namespace qtgui */
} /* namespace gr */
