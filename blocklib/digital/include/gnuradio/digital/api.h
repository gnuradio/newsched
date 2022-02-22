/*
 * Copyright 2011 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/attributes.h>

#ifdef gnuradio_digital_EXPORTS
#define DIGITAL_API __GR_ATTR_EXPORT
#else
#define DIGITAL_API __GR_ATTR_IMPORT
#endif
